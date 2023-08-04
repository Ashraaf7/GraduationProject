using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;
using System.Web.Mvc;

namespace CourseApp1.data
{
    [MetadataType(typeof(CategoryeMetadata))]
    public partial class Category
    {
        public static implicit operator Category(SelectList v)
        {
            throw new NotImplementedException();
        }
    }
    public class CategoryeMetadata
    {
        public int ID { get; set; }
        [Display(Name = "Category")]
        public string Name { get; set; }
        public Nullable<int> Parent_id { get; set; }
    }
}